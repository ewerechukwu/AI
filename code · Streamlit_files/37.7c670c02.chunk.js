/*! For license information please see 37.7c670c02.chunk.js.LICENSE.txt */
(this["webpackJsonpstreamlit-browser"]=this["webpackJsonpstreamlit-browser"]||[]).push([[37],{4046:function(t,e,s){"use strict";s.r(e),s.d(e,"default",(function(){return l}));var i=s(0),n=s.n(i),a=s(205),o=s(5);class r extends n.a.PureComponent{constructor(...t){super(...t),this.state={value:this.initialValue},this.setWidgetValue=t=>{const e=this.props.element.id;this.props.widgetMgr.setIntValue(e,this.state.value,t)},this.onChange=t=>{this.setState({value:t},(()=>this.setWidgetValue({fromUi:!0})))},this.render=()=>{const t=this.props.element,e=t.options,s=t.help,i=t.label,n=this.props.disabled;return Object(o.jsx)(a.b,{label:i,options:e,disabled:n,width:this.props.width,onChange:this.onChange,value:this.state.value,help:s})}}get initialValue(){const t=this.props.element.id,e=this.props.widgetMgr.getIntValue(t);return void 0!==e?e:this.props.element.default}componentDidMount(){this.setWidgetValue({fromUi:!1})}}var l=r}}]);
//# sourceMappingURL=37.7c670c02.chunk.js.map